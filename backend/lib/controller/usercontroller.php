<?php
/*
 * Copyright (c) 2010 by Justin Otherguy <justin@justinotherguy.org>
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (either version 2 or
 * version 3) as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 * For more information on the GPL, please go to:
 * http://www.gnu.org/copyleft/gpl.html
 */

class UserController extends Controller {
	public function add() {
		$user = new User();
		$user->password = $this->view->request->get['password'];
		
		// TODO generate uuid
		
		$user->save();
		$this->view->addUser($user);
	}
	
	public function delete() {
		$user = User::getByUuid($this->view->request->get['uuid']);
		$user->delete();
	}
	
	public function edit() {
		// TODO implement UserController::edit();
	}
	
	public function subscribe() {
		// TODO implement UserController::subscribe();
	}
	
	public function unsubscribe() {
		// TODO implement UserController::unsubscribe();
	}
}